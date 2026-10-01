# S4 runaway investigation

Verdict: **UNRESOLVED for the campaign configuration.** No CPU reproduction identifies a k=4/subset defect. A separate **BUG** in checkpoint replay is reproduced, but ordinary campaign rejection cannot reach that path under the source's configured rollback window. Calling the campaign result EXACT-LIKELY-CHANCE would exceed the evidence.

Labels: M = measured/observed; D = derived/inferred; G = guess. Parameter settings and identifiers are literal configuration, not estimates. Source locations refer to `0dba27b9002b8347c6f14a9f4a1651b3478065ef` unless marked as harness code.

## Scope and design

Shared local clone: `/home/xanewok/repos/s4-runaway/stack2`, topic branch `codex/s4-runaway`. The campaign source is `68d99c67d` plus host input uploads, backend verify candidates, reduced draft vocabulary, small-subset guards, and sampler replacement/detachment fixes. Patch A's multi-entry graph cache is absent. Production code is unchanged; the existing TP sampling test now covers verify rows for the requested depths. Harnesses and evidence are under `stack2/codex`; the outer `codex/REPORT.md` points here.

Read the supplied GPU sampling reports, Patch A report/review, CPU fixture launcher, frozen queue script, and prior subset/verify harnesses before designing checks. Campaign data and the weekend repository were read only. All executed binaries were CPU builds, with CUDA hidden, nice 19, and at most eight compute threads per process; Meta CPU contexts use four threads per device. The only server executed was the renamed `s4-cpu-srv`, stopped by its exact child PID. No GPU query, CUDA binary, remote API, push, or download was executed successfully.

Deviation: the initial server build invoked CMake's automatic UI download hook. DNS failed before any download; this was an unintended network attempt. Subsequent builds set `LLAMA_USE_PREBUILT_UI=OFF`, `LLAMA_BUILD_UI=OFF`, `LLAMA_CURL=OFF`, and `GGML_CCACHE=OFF`. The initial ccache write also failed outside the worktree. Early compiler invocations did not explicitly constrain temporary-file placement; the reproducible CPU launcher sets TMPDIR inside build-cpu. The CPU HTTP fixture needed sandbox escalation for its localhost socket; it then completed.

## Source audit

| Mechanism | Evidence and conclusion |
|---|---|
| Recurrent rollback window | `common/common.h:400`, `common/common.cpp:1637`: MTP sets `n_rs_seq=draft.n_max`. `src/models/delta-net-base.cpp:563` allocates D: `k+1` snapshot groups, including current state. `server-context.cpp:4071` computes `n_rollback=draft.size()+1-accepted.size()`, bounded by draft size because at least one corrected/bonus token is returned. Thus D: maximum rollback is k, not k+1. `common/speculative.cpp:1761` caps generated proposals at configured n_max. No k=4 overflow follows from the extra verify row. |
| Snapshot indexing | `src/llama-memory-recurrent.cpp:193`, `src/llama-memory-recurrent.cpp:1350`; `src/models/delta-net-base.cpp:449` and `:527`. Conv and GDN snapshots are addressed by rollback distance. `ggml/src/ggml-cuda/gated_delta_net.cu:149` loops over snapshot slots without a depth-four special case. CPU state replay checks pass for every rollback distance. GPU execution remains unverified. |
| Checkpoint selection | `server-context.cpp:3204`, `:4075`: checkpoint/restore is selected when the draft/rollback exceeds the window, or removal type is FULL. `common/common.cpp:1497` returns RS when n_rs_seq is positive. D: Qwen3.5 MTP with global k and the same requested draft maximum stays within the RS path; shrinking the final round only reduces rollback. See the confirmed replay bug below. |
| Verify top-K certificate | `common/sampling.cpp:156`, `:561`, `:594`: target top_k=20 requests D: 21 candidates per row. It is independent of speculative k. Finite, strictly ordered candidates certify selection; ties/non-finite cutoff fetch the full row. CPU host/candidate seed identity passes for all requested depths. |
| Subset scatter and outside tokens | `src/llama-context.cpp:1308` enables compact logits only for an MTP context. `:1312` scatters by real token ID, fills excluded IDs with -inf, and clears each pending flag. `:913` reads compact dimensions before scatter. `common/speculative.cpp:1728` records probabilities of the actual sampled proposal. Target logits and token embeddings retain the full vocabulary; corrected/bonus IDs do not index the compact output head. CPU corrected tokens outside the fixture subset, repeated/deferred reads and multiple microbatches pass. |
| Rejection law | `common/sampling.cpp:936`: accept mass is min(p,q); residual mass is max(p-q,0), including target tokens outside q. For normalized actual q, their sum is p. Backend target and draft chains only compute candidates; the host draws and records q. This argument assumes the RNG draws remain independent of the already emitted history, which checkpoint replay violates. |
| Ordinary RNG consumption | `common/speculative.cpp:66` mixes the explicit draft seed; `:1677` avoids resetting probabilistic draft samplers each round. Target acceptance has its own stream (`common/sampling.cpp:126`, `:585`). Different k/subsets consume different random streams; probabilistic seed identity versus non-spec decoding is not required. Greedy non-spec identity is applicable and passes. |
| Backend binding and reserve | `src/llama-context.cpp:1387` invalidates graph bindings on replacement; `:1527` only reuses the active graph result. This avoids the reviewed Patch A cross-graph binding case. `server-context.cpp:1101` uses output limits computed at `common/speculative.cpp:2674` from n_max+1. `src/llama-context.cpp:2540` budgets sampler nodes for multiple output rows, including candidate gather nodes. TP candidate/deferred-logit tests pass at all requested row counts. |
| GPU batch boundary | `ggml/src/ggml-cuda/fattn.cu:156` and `:204`: q8_0 MMA uses ncols1=4 for four verify queries, ncols1=8 for five/six. D: k=4 crosses that dispatch boundary, although k=5 shares its tile width. Existing backend-op cases include these shapes (`tests/test-backend-ops.cpp:11107`); none were executed here. CUDA graphs, BF16 internal AllReduce and asynchronous input uploads are also outside CPU evidence. No concrete defect was established in these paths. |

## Confirmed checkpoint replay RNG bug

**History-dependent acceptance draw reuse.** `tools/server/server-context.cpp:4101` restores the pre-verify sampler after retaining the accepted/corrected tokens. `common/sampling.cpp:707` copies the acceptance RNG back; `server_accept_replay` at `server-context.cpp:58` advances the probability-chain RNG but never advances the separate acceptance RNG. The next verify can reuse a uniform conditioned by the retained history.

CPU repro: `codex/check.cpp:181` invokes the actual rejection sampler and copy API, then reproduces the replay helper's sample/accept sequence. Let target p(A)=p(B)=1/2 and draft q(A)=1. Condition on first-position rejection: the retained token is B and the acceptance uniform U is at least 1/2. After checkpoint rewind and replay, the next draft A is rejected with the same U, so its emitted token is always B; the target conditional law requires half A. Keeping the post-verify acceptance RNG is the positive control; this is an oracle control, not a production fix.

| Configured depth | Conditioned observations (M) | Next A,B with rewind (M) | Chi-square, df=1 (M) | A,B with preserved RNG (M) |
|---|---:|---:|---:|---:|
| k=3 | 9,966 | 0, 9,966 | 9,966 | 4,969, 4,997 |
| k=4 | 9,974 | 0, 9,974 | 9,974 | 4,983, 4,991 |
| k=5 | 9,974 | 0, 9,974 | 9,974 | 4,985, 4,989 |

M: each experiment starts with 20,000 rounds; preserved-RNG chi-square is at most 0.079. Confidence: high for the replay-path defect, low for any connection to this campaign. D: the campaign's usual RS rejection never restores a speculative checkpoint. Its logs do not expose checkpoint counters, so absence of messages alone cannot prove path absence. Do not fix replay by retaining the entire post-verify sampler and then re-accepting tokens: that would double-advance grammar/penalty/budget history. A fix must preserve consumed random state while restoring and replaying only the history state; it is outside this investigation.

## Campaign observations

Campaign root: `/home/xanewok/repos/qwen38-weekend/results/weekend-20260930-main/`. All table values below are M from `probe.json` or `server.log`. M: all alphametics attempts share prompt-0 SHA256 prefix `10a9c98bb3` and template SHA256 prefix `4af3dcf0e9`. Launches use the same frozen source/binary configuration apart from k/subset. The recorded harness seed is 1729; it is not proof of a common server sampling seed because launch has no explicit --seed and request payloads are not archived here.

| Arm | Completion tokens by pair (M) |
|---|---|
| S0, full k3 | 8,392 / 11,153 / 10,371 |
| S3, subset k3 | 9,063 / 6,175 / 3,761 |
| S4, subset k4 | 121,806 / 82,848 / 108,747 |
| S5, subset k5 | 7,592 / 20,604 / 6,184 |

S4 attempt ID is `0957923a4a4b4e1b-01`; the campaign directory distinguishes pairs. The long response occurs after the initial read/explore tool calls, rather than after an unusually large initial prompt. The first request processes M: 6,428 prompt tokens in every arm.

| S4 pair | Long response task | Newly evaluated prompt tokens (M) | Long response tokens (M) | Draft acceptance (M) | End context / truncated (M) | server.log lines |
|---|---:|---:|---:|---:|---|---|
| pair1 | 61 | 1,594 | 93,872 | 0.94990 | 102,399 / 1 | 282, 283, 286, 287 |
| pair2 | 72 | 1,875 | 79,044 | 0.91734 | 87,924 / 0 | 243, 244, 247, 248 |
| pair3 | 67 | 1,869 | 93,611 | 0.94723 | 102,399 / 1 | 279, 280, 283, 284 |

M: later resumed responses have acceptance 0.55912 / 0.57228 / 0.61935 and lengths 24,620 / 2,479 / 14,145. Thus the high task-level acceptance is concentrated in the long response. Saved reasoning in `codex-state/sessions` confirms comma-separated letter repetition in each initial S4 long response; details are recorded below.

| S4 pair | Reasoning characters (M) | Longest continuous letter/punctuation span (M) | Character offset of that span (M) | Rollout filename and line |
|---|---:|---:|---:|---|
| pair1 | 153,290 | 129,905 | 23,385 | `rollout-2026-09-30T23-22-37-01a0f4a0-8b1d-73a1-8a1b-f316dd232125.jsonl:26` |
| pair2 | 140,318 | 102,132 | 38,185 | `rollout-2026-10-01T00-51-09-01a0f4f1-9b77-7fa1-9b00-bd89e1b6d1dc.jsonl:28` |
| pair3 | 151,649 | 131,066 | 20,583 | `rollout-2026-10-01T03-20-10-01a0f57a-06b9-7352-a17b-cc35bd5134ec.jsonl:31` |

M: the regex `(?:[A-Z][,.] ?){16,}` finds no such span in the other alphametics attempts. The S4 tails repeat sequences such as `S, S, S, E` and `S, F, O, R, T, H, E, F, I, R, E`. `codex/CAMPAIGN.txt` records full rollout paths and samples; character offsets are not original token positions. This confirms a reasoning repetition failure, while its cause remains unresolved.

M: no restore/replay/checkpoint messages or error-level lines occur in any phase1 attempt. Verbosity is 3, with no speculative trace flag, so routine checkpoint debug logs and trace-gated acceptance logs would be suppressed. Shared warnings concern unset API key, tensor-mode memory fitting, NCCL fallback, reasoning preservation and skipped namespace tools. Only S4 pair1 adds cancellation of task 19993, after the long response has already ended; it cannot explain onset.

D: pooled acceptance excluding alphametics is S0=0.7635, S3=0.7228, S4=0.6620, S5=0.5981, from M: 24 attempts per arm. There is no pooled k4 discontinuity on the other tasks. D: the nominal Fisher probability is 1/220 = 0.004545 under the stated independent-arm null. That association does not establish a changed output law, and task selection, request histories, actual server seeds and GPU numerical behavior are not controlled by that calculation. `codex/campaign.py` regenerates the read-only extraction.

## CPU evidence and limitations

Fixture: `/home/xanewok/models/Qwen3.5-4B-UD-Q4_K_XL.gguf`. The fixture subset is a descending every-tenth ID list from this model's own vocabulary, M: 24,832 IDs. It is not the Swift campaign's different M: 25,879-ID list. `codex/EVIDENCE.txt` records measured results; raw local logs are also retained.

- M: 256 host/candidate seed identities per k/subset configuration. This compares the same proposals and p rows, not probabilistic spec versus non-spec streams.
- M: 24 next-token law checks, each with 20,000 draws, exercise every verify position at k=3,4,5. Earlier rows use q=p so the position is reached; at the tested row q is normalized and either unrestricted/tilted or restricted. Real accept/reject and outside-subset residual draws are exercised. D: exact chi-square survival probabilities have minimum 0.07222, df=18. Deliberately recording one-tenth of actual q gives M: chi-square above 15,000. `S4_CORRUPT_Q=1` makes the normal oracle exit 1 with `FAIL: law oracle failed` (M: chi-square 3703.95).
- M: 24 forced partial rollbacks per backend mode, with every distance from one through configured k, full/subset models, dirty checkpoint restores, q8_0 target KV and a corrected token outside the subset. All final logits are bit-identical to the independently evaluated retained-prefix reference. Leaving the rejected suffix in state is detected; M: NMSE controls are roughly 0.15. After the maximum rollback at each configuration, M: 20,000 real-row sampling draws pass the target-law check; D: minimum p is 0.07052 across single/Meta CPU modes. These proposals are constructed from the target support; they are not an end-to-end MTP distribution test.
- M: real-model single-CPU verify rows use installed backend candidates and match host seed draws. The 4B model's tied target output fails the Meta TP backend gate, so the Meta real-model state fixture uses the host fallback. The separate untied synthetic TP model covers actual Meta backend candidates, full deferred logits, replacement/detachment and multi-microbatch lifetime: M: 64 seeds per row, four/five/six rows, all pass.
- M: the adapted prior reduced-head fixture passes at four/five/six output rows on CPU and Meta CPU: target bit identity, raw compact/full row equality, excluded masking, normalized actual draft q, real token IDs, backend global top-k, deferred readback and small-subset guards. Maximum selected-row logit error is M: zero in these runs. Wrong token-ID mapping is detected.
- M: renamed CPU server outputs match non-spec greedy decoding across every k/subset arm, 48 predicted tokens per boot; digest `4a9eabc8202b24947a03dcd259ef376bd5d0a75453e0ddd9732ffbeea1b97106`. Every spec arm produced draft tokens.

The initial optimized CPU build had a batch-dependent mismatch before rollback: M: prefix NMSE 0.001159784 and post-rollback NMSE 0.001305671 at k3/full. Disabling `GGML_CPU_REPACK` eliminated it and made all tested prefixes/rollback outputs bit-identical. D: this localizes the mismatch to the repacked CPU computation path, not an established rollback defect; it does not explain the CUDA campaign. CPU checks use the explicit reference setting `GGML_CPU_REPACK=OFF`. No performance conclusion is drawn from the two-step head timing logs or from loaded-host timings.

Reproduce with `CUDA_VISIBLE_DEVICES='' nice -n 19 bash codex/run_cpu.sh`; it configures `cmake -B build-cpu -DGGML_CUDA=OFF`, uses -j8, disables network build hooks, compiles the standalone harnesses and runs the CPU fixtures. Existing `test-sampling` and the broader grammar/prefilter suite pass with M: zero failures; the latter reproduces stream hash `e97ea9873285a3f9`. TP sampling passes at all requested depths.

## Parent-only GPU follow-up

`codex/gpu_ab.sh` defaults to plan mode and was not GPU-executed. Its shell syntax, dry plan, parent-enabled probe compilation against CPU libraries, and comparison oracle controls pass. The oracle accepts equal binary rows and fails a perturbed row. No new CUDA build was needed because production code is unchanged and the probe links the parent's already frozen stack2 libraries.

After **2026-10-03 06:00 Europe/Warsaw** (timezone assumption from the task environment), parent command: `RUN_PARENT_GPU=1 PLAN=0 nice -n 19 bash codex/gpu_ab.sh`.

This is a mechanism test, not a speed benchmark: identical frozen bytes, interleaved up/patch/patch/up process boots, switch `GGML_CUDA_FA_MMA_Q8_0=0/1`, teacher-forced identical Swift histories, k3/k4/k5, full/campaign subset, q8_0 KV, tensor split and actual backend candidates. It records full F32 verify, post-rollback and reference rows for cross-arm/repeated-boot comparison. The script freezes the probe and libraries, checks manifests/model/subset hashes and exact source, refuses busy devices or existing serving processes, checks Xid, enforces a configured 420-second wall budget and stops only its owned child PID. It never starts a process named llama-server.

A row difference under the FA switch localizes a numerical/integration problem before another long campaign. A clean result rules out that switch on these short histories; it cannot certify long-context attention, every CUDA graph/input-upload schedule, or the entire agentic law. Next smallest extension would reconstruct a teacher-forced prefix near repetition onset from the saved rollouts, then compare logits on identical tokenized histories before allowing generated trajectories to diverge. The inspected files retain reasoning text but not original server token IDs or raw request payloads. Checkpoint RNG repair remains a separate follow-up, independent of the S4 verdict.
